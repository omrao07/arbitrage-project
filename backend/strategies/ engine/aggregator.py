# backend/engine/aggregator.py

import pandas as pd
from ..utils.logger import log

class Aggregator:
    def __init__(self):
        self.signals = []
        self.weights = []
        self.pnls = []

    def collect_signal(self, strategy_name, signal_df):
        log(f"[Aggregator] Collected signal for {strategy_name}")
        self.signals.append((strategy_name, signal_df))

    def collect_weight(self, strategy_name, weight):
        log(f"[Aggregator] Collected weight for {strategy_name}: {weight}")
        self.weights.append((strategy_name, weight))

    def collect_pnl(self, strategy_name, pnl_df):
        log(f"[Aggregator] Collected PnL for {strategy_name}")
        self.pnls.append((strategy_name, pnl_df))

    def aggregate_signals(self):
        if not self.signals:
            return pd.DataFrame()

        log("[Aggregator] Aggregating signals...")
        merged_df = pd.DataFrame()

        for name, df in self.signals:
            df = df.copy()
            df.columns = [f"{name}_{col}" for col in df.columns]
            merged_df = pd.concat([merged_df, df], axis=1)

        return merged_df

    def aggregate_weights(self):
        if not self.weights:
            return {}

        log("[Aggregator] Aggregating weights...")
        return dict(self.weights)

    def aggregate_pnls(self):
        if not self.pnls:
            return pd.DataFrame()

        log("[Aggregator] Aggregating PnLs...")
        pnl_df = pd.DataFrame()
        for name, df in self.pnls:
            df = df.copy()
            df.columns = [f"{name}_pnl" for col in df.columns]
            pnl_df = pd.concat([pnl_df, df], axis=1)

        return pnl_df

    def reset(self):
        log("[Aggregator] Resetting aggregator...")
        self.signals = []
        self.weights = []
        self.pnls = []