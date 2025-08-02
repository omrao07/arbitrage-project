# region_router.py

from backend.engine.strategy_router import StrategyRouter
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class RegionRouter:
    def __init__(self, region_models: dict):
        """
        region_models: Dictionary of region -> StrategyRouter instance.
        Example:
        {
            'us': StrategyRouter(...),
            'india': StrategyRouter(...),
            ...
        }
        """
        self.region_models = region_models
        self.global_weights = {
            'us': 0.30,
            'india': 0.20,
            'china': 0.15,
            'europe': 0.15,
            'japan': 0.10,
            'global': 0.10  # global macro overlay
        }

    def generate_global_signals(self, market_data):
        global_signals = {}
        for region, router in self.region_models.items():
            try:
                logger.info(f"Routing strategies for {region}")
                signals = router.generate_signals(market_data)
                weighted_signals = {
                    s: v * self.global_weights[region]
                    for s, v in signals.items()
                }
                for symbol, value in weighted_signals.items():
                    global_signals[symbol] = global_signals.get(symbol, 0) + value
            except Exception as e:
                logger.warning(f"[{region}] router failed: {e}")
        return global_signals

    def get_region_signal_breakdown(self, market_data):
        """
        Returns signal breakdown from all regions individually.
        """
        breakdown = {}
        for region, router in self.region_models.items():
            try:
                signals = router.generate_signals(market_data)
                breakdown[region] = signals
            except Exception as e:
                logger.warning(f"[{region}] signal generation failed: {e}")
                breakdown[region] = {}
        return breakdown