// config/defaults.ts
// Default engine configuration (no imports, pure Node)

export const defaults = {
  engine: {
    name: "Adaptive Engine",
    mode: "development",       // development | production
    logLevel: "info",          // debug | info | warn | error
    tickIntervalMs: 1000       // default scheduler tick = 1s
  },

  risk: {
    maxLeverage: 5,
    maxDrawdownPct: 0.2,       // 20%
    perStrategyLimitUSD: 1e6,  // $1m
    globalLimitUSD: 1e7        // $10m
  },

  portfolio: {
    baseCurrency: "USD",
    rebalanceInterval: "1d",   // daily rebalance
    targetVolatility: 0.15     // 15% annualized
  },

  data: {
    cacheTTL: 60,              // seconds
    retryCount: 3,
    retryDelayMs: 500
  },

  connectors: {
    broker: {
      ibkr: { enabled: false },
      zerodha: { enabled: false },
      alpaca: { enabled: false }
    },
    exchange: {
      binance: { enabled: false },
      cme: { enabled: false },
      nse: { enabled: false }
    }
  }
};