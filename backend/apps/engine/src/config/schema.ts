// config/schema.ts
// Type-only schema + tiny runtime validators (pure TS, no imports)

/* ============================== Types ============================== */

export type EngineConfig = {
  name: string;               // e.g., "Adaptive Engine"
  mode: "development" | "production";
  logLevel: "debug" | "info" | "warn" | "error";
  tickIntervalMs: number;     // main scheduler tick in ms
};

export type RiskConfig = {
  maxLeverage: number;        // e.g., 5
  maxDrawdownPct: number;     // 0..1 (e.g., 0.2 = 20%)
  perStrategyLimitUSD: number;
  globalLimitUSD: number;
};

export type PortfolioConfig = {
  baseCurrency: string;       // e.g., "USD"
  rebalanceInterval: string;  // e.g., "1d", "1h"
  targetVolatility: number;   // 0..1 (annualized)
};

export type DataConfig = {
  cacheTTL: number;           // seconds
  retryCount: number;         // integer >= 0
  retryDelayMs: number;       // ms
};

export type ConnectorsToggle = { enabled: boolean };

export type ConnectorsConfig = {
  broker: {
    ibkr: ConnectorsToggle;
    zerodha: ConnectorsToggle;
    alpaca: ConnectorsToggle;
  };
  exchange: {
    binance: ConnectorsToggle;
    cme: ConnectorsToggle;
    nse: ConnectorsToggle;
  };
};

export type EngineConfigRoot = {
  engine: EngineConfig;
  risk: RiskConfig;
  portfolio: PortfolioConfig;
  data: DataConfig;
  connectors: ConnectorsConfig;
};

/* =========================== Type Guards ============================ */

export function isEngineConfig(x: any): x is EngineConfig {
  return !!x
    && typeof x.name === "string"
    && (x.mode === "development" || x.mode === "production")
    && (x.logLevel === "debug" || x.logLevel === "info" || x.logLevel === "warn" || x.logLevel === "error")
    && isFiniteNumber(x.tickIntervalMs, 1);
}

export function isRiskConfig(x: any): x is RiskConfig {
  return !!x
    && isFiniteNumber(x.maxLeverage)
    && isFiniteNumber(x.maxDrawdownPct, 0, 1)
    && isFiniteNumber(x.perStrategyLimitUSD, 0)
    && isFiniteNumber(x.globalLimitUSD, 0);
}

export function isPortfolioConfig(x: any): x is PortfolioConfig {
  return !!x
    && typeof x.baseCurrency === "string"
    && typeof x.rebalanceInterval === "string"
    && isFiniteNumber(x.targetVolatility, 0, 1);
}

export function isDataConfig(x: any): x is DataConfig {
  return !!x
    && isFiniteNumber(x.cacheTTL, 0)
    && isFiniteInteger(x.retryCount, 0)
    && isFiniteNumber(x.retryDelayMs, 0);
}

export function isConnectorsConfig(x: any): x is ConnectorsConfig {
  const ok = (t: any) => !!t && typeof t.enabled === "boolean";
  return !!x
    && x.broker && ok(x.broker.ibkr) && ok(x.broker.zerodha) && ok(x.broker.alpaca)
    && x.exchange && ok(x.exchange.binance) && ok(x.exchange.cme) && ok(x.exchange.nse);
}

export function isEngineConfigRoot(x: any): x is EngineConfigRoot {
  return !!x
    && isEngineConfig(x.engine)
    && isRiskConfig(x.risk)
    && isPortfolioConfig(x.portfolio)
    && isDataConfig(x.data)
    && isConnectorsConfig(x.connectors);
}

/* ========================= Runtime Validation ======================= */

export type ValidationResult =
  | { ok: true; data: EngineConfigRoot }
  | { ok: false; errors: string[] };

/**
 * validateConfig
 * - Checks the shape and constraints of a config object.
 * - Returns { ok, data } on success or { ok:false, errors } on failure.
 * - Does NOT mutate input; does NOT apply defaults (pass an already-merged object).
 */
export function validateConfig(input: any): ValidationResult {
  const errors: string[] = [];
  const path = (k: string) => `config.${k}`;

  if (!input || typeof input !== "object") {
    return { ok: false, errors: ["config: must be an object"] };
  }

  // engine
  if (!isEngineConfig(input.engine)) {
    errors.push(`${path("engine")}: invalid`);
  }

  // risk
  if (!isRiskConfig(input.risk)) {
    errors.push(`${path("risk")}: invalid`);
  }

  // portfolio
  if (!isPortfolioConfig(input.portfolio)) {
    errors.push(`${path("portfolio")}: invalid`);
  }

  // data
  if (!isDataConfig(input.data)) {
    errors.push(`${path("data")}: invalid`);
  }

  // connectors
  if (!isConnectorsConfig(input.connectors)) {
    errors.push(`${path("connectors")}: invalid`);
  }

  if (errors.length) return { ok: false, errors };
  // as EngineConfigRoot because we've verified via guards
  return { ok: true, data: input as EngineConfigRoot };
}

/* ============================ Normalization ========================== */

/**
 * normalizeConfig
 * - Shallowly fills missing fields from `fallbacks` (typically your defaults).
 * - Then validates. Returns either { ok:true, data } or { ok:false, errors }.
 */
export function normalizeConfig(partial: any, fallbacks: EngineConfigRoot): ValidationResult {
  const merged: EngineConfigRoot = {
    engine: {
      name: pickStr(partial?.engine?.name, fallbacks.engine.name),
      mode: pickUnion(partial?.engine?.mode, ["development", "production"], fallbacks.engine.mode),
      logLevel: pickUnion(partial?.engine?.logLevel, ["debug", "info", "warn", "error"], fallbacks.engine.logLevel),
      tickIntervalMs: pickNum(partial?.engine?.tickIntervalMs, fallbacks.engine.tickIntervalMs),
    },
    risk: {
      maxLeverage: pickNum(partial?.risk?.maxLeverage, fallbacks.risk.maxLeverage),
      maxDrawdownPct: clampNum(pickNum(partial?.risk?.maxDrawdownPct, fallbacks.risk.maxDrawdownPct), 0, 1),
      perStrategyLimitUSD: pickNum(partial?.risk?.perStrategyLimitUSD, fallbacks.risk.perStrategyLimitUSD),
      globalLimitUSD: pickNum(partial?.risk?.globalLimitUSD, fallbacks.risk.globalLimitUSD),
    },
    portfolio: {
      baseCurrency: pickStr(partial?.portfolio?.baseCurrency, fallbacks.portfolio.baseCurrency),
      rebalanceInterval: pickStr(partial?.portfolio?.rebalanceInterval, fallbacks.portfolio.rebalanceInterval),
      targetVolatility: clampNum(pickNum(partial?.portfolio?.targetVolatility, fallbacks.portfolio.targetVolatility), 0, 1),
    },
    data: {
      cacheTTL: Math.max(0, pickNum(partial?.data?.cacheTTL, fallbacks.data.cacheTTL)),
      retryCount: Math.max(0, Math.floor(pickNum(partial?.data?.retryCount, fallbacks.data.retryCount))),
      retryDelayMs: Math.max(0, pickNum(partial?.data?.retryDelayMs, fallbacks.data.retryDelayMs)),
    },
    connectors: {
      broker: {
        ibkr: { enabled: pickBool(partial?.connectors?.broker?.ibkr?.enabled, fallbacks.connectors.broker.ibkr.enabled) },
        zerodha: { enabled: pickBool(partial?.connectors?.broker?.zerodha?.enabled, fallbacks.connectors.broker.zerodha.enabled) },
        alpaca: { enabled: pickBool(partial?.connectors?.broker?.alpaca?.enabled, fallbacks.connectors.broker.alpaca.enabled) },
      },
      exchange: {
        binance: { enabled: pickBool(partial?.connectors?.exchange?.binance?.enabled, fallbacks.connectors.exchange.binance.enabled) },
        cme:     { enabled: pickBool(partial?.connectors?.exchange?.cme?.enabled,     fallbacks.connectors.exchange.cme.enabled) },
        nse:     { enabled: pickBool(partial?.connectors?.exchange?.nse?.enabled,     fallbacks.connectors.exchange.nse.enabled) },
      },
    },
  };

  return validateConfig(merged);
}

/* ============================== Helpers ============================== */

function isFiniteNumber(x: any, min?: number, max?: number): boolean {
  const n = Number(x);
  if (!Number.isFinite(n)) return false;
  if (min !== undefined && n < min) return false;
  if (max !== undefined && n > max) return false;
  return true;
}

function isFiniteInteger(x: any, min?: number, max?: number): boolean {
  const n = Number(x);
  if (!Number.isFinite(n) || Math.floor(n) !== n) return false;
  if (min !== undefined && n < min) return false;
  if (max !== undefined && n > max) return false;
  return true;
}

function pickStr(v: any, fallback: string): string {
  return typeof v === "string" && v.length > 0 ? v : fallback;
}

function pickNum(v: any, fallback: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function pickBool(v: any, fallback: boolean): boolean {
  return typeof v === "boolean" ? v : fallback;
}

function pickUnion<T extends string>(v: any, allowed: readonly T[], fallback: T): T {
  return (allowed as readonly string[]).includes(v) ? (v as T) : fallback;
}

function clampNum(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n));
}