// config/env.ts
// Merge process.env values into defaults (pure Node)

import { defaults } from "./defaults.js";

export function loadConfig() {
  const cfg = JSON.parse(JSON.stringify(defaults)); // deep clone

  try {
    if (process.env.ENGINE_MODE) cfg.engine.mode = process.env.ENGINE_MODE;
    if (process.env.ENGINE_LOG_LEVEL) cfg.engine.logLevel = process.env.ENGINE_LOG_LEVEL;
    if (process.env.ENGINE_TICK_MS) cfg.engine.tickIntervalMs = parseInt(process.env.ENGINE_TICK_MS, 10);

    if (process.env.RISK_MAX_LEVERAGE) cfg.risk.maxLeverage = Number(process.env.RISK_MAX_LEVERAGE);
    if (process.env.RISK_MAX_DRAWDOWN) cfg.risk.maxDrawdownPct = Number(process.env.RISK_MAX_DRAWDOWN);
    if (process.env.RISK_PER_STRAT_USD) cfg.risk.perStrategyLimitUSD = Number(process.env.RISK_PER_STRAT_USD);
    if (process.env.RISK_GLOBAL_USD) cfg.risk.globalLimitUSD = Number(process.env.RISK_GLOBAL_USD);

    if (process.env.PORTFOLIO_BASE) cfg.portfolio.baseCurrency = process.env.PORTFOLIO_BASE;
    if (process.env.PORTFOLIO_REBALANCE) cfg.portfolio.rebalanceInterval = process.env.PORTFOLIO_REBALANCE;
    if (process.env.PORTFOLIO_VOL) cfg.portfolio.targetVolatility = Number(process.env.PORTFOLIO_VOL);

    if (process.env.DATA_CACHE_TTL) cfg.data.cacheTTL = Number(process.env.DATA_CACHE_TTL);
    if (process.env.DATA_RETRY_COUNT) cfg.data.retryCount = Number(process.env.DATA_RETRY_COUNT);
    if (process.env.DATA_RETRY_DELAY) cfg.data.retryDelayMs = Number(process.env.DATA_RETRY_DELAY);

    // Connector toggles
    ["ibkr","zerodha","alpaca"].forEach((b) => {
      const key = "BROKER_" + b.toUpperCase();
      if (process.env[key]) cfg.connectors.broker[b].enabled = process.env[key] === "true";
    });
    ["binance","cme","nse"].forEach((x) => {
      const key = "EXCHANGE_" + x.toUpperCase();
      if (process.env[key]) cfg.connectors.exchange[x].enabled = process.env[key] === "true";
    });
  } catch {
    // swallow any parsing errors
  }

  return cfg;
}