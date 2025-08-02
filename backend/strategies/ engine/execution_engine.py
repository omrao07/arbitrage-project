import datetime
import logging
import random
from typing import Dict, List

class ExecutionEngine:
    def __init__(self):
        self.orders_executed = []
        self.position_book = {}
        self.logger = logging.getLogger("ExecutionEngine")

    def execute_signals(self, aggregated_signals: Dict[str, Dict[str, float]]) -> None:
        """
        Execute trades based on aggregated signals. Each strategy's weights are used
        to calculate order sizes. Execution is mocked here.
        """
        for strategy, signals in aggregated_signals.items():
            for asset, weight in signals.items():
                price = self.mock_market_price(asset)
                quantity = round(weight * 1000 / price, 2)  # Assume fixed capital per trade
                trade = {
                    'timestamp': datetime.datetime.utcnow().isoformat(),
                    'strategy': strategy,
                    'asset': asset,
                    'price': price,
                    'quantity': quantity,
                    'action': 'BUY' if weight > 0 else 'SELL'
                }
                self.orders_executed.append(trade)
                self.update_position(asset, quantity, weight > 0)
                self.logger.info(f"Executed trade: {trade}")

    def update_position(self, asset: str, quantity: float, is_buy: bool) -> None:
        """Update current position book based on executed trade."""
        if asset not in self.position_book:
            self.position_book[asset] = 0
        self.position_book[asset] += quantity if is_buy else -quantity

    def get_position_book(self) -> Dict[str, float]:
        return self.position_book

    def get_executed_orders(self) -> List[Dict]:
        return self.orders_executed

    @staticmethod
    def mock_market_price(asset: str) -> float:
        """Mock function to simulate asset price."""
        return round(random.uniform(80, 300), 2)