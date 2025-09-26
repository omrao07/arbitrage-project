// risk/limits.ts
// Import-free risk limits module.
//
// Provides a registry of per-strategy and global risk limits,
// checks against those limits, and emits violations.
//
// Limits supported:
//   - maxPosition (abs qty per symbol)
//   - maxNotional (abs $ per symbol)
//   - maxLoss (cumulative realized loss)
//   - maxDrawdown (fraction of peak equity)
//   - maxGrossExposure (Σ |positions| vs equity)
//   - maxLeverage (gross / equity)
//   - maxOrdersPerMin (rate limiter)
//
// Usage:
//   const rl = createRiskLimits({ equity: 1_000_000 });
//   rl.set("global", { maxLeverage: 3, maxLoss: -200000 });
//   rl.set("AAPL", { maxPosition: 5000, maxNotional: 1_000_000 });
//   const r = rl.check({ symbol: "AAPL", qty: 6000, notional: 1_200_000 });
//   if (!r.ok) console.log(r.violations);

export type LimitConfig = {
  maxPosition?: number;
  maxNotional?: number;
  maxLoss?: number;
  maxDrawdown?: number;      // 0.2 = 20%
  maxGrossExposure?: number; // in $
  maxLeverage?: number;      // gross/equity
  maxOrdersPerMin?: number;
};

export type Violation = {
  key: string;          // symbol or "global"
  field: keyof LimitConfig;
  value: number;
  limit: number;
};

export type CheckInput = {
  symbol?: string;
  qty?: number;
  notional?: number;
  equity?: number;
  gross?: number;
  realized?: number;
  peakEquity?: number;
  ordersLastMin?: number;
};

export type CheckResult = {
  ok: boolean;
  violations: Violation[];
};

export type RiskLimitsAPI = {
  set(key: string, cfg: LimitConfig): void;
  get(key: string): LimitConfig | undefined;
  all(): Record<string, LimitConfig>;
  check(i: CheckInput): CheckResult;
  clear(): void;
};

export function createRiskLimits(init: { equity?: number } = {}): RiskLimitsAPI {
  const limits = new Map<string, LimitConfig>();
  let equity = num(init.equity, 0);

  function set(key: string, cfg: LimitConfig) {
    limits.set(key, { ...cfg });
  }

  function get(key: string) {
    return limits.get(key);
  }

  function all() {
    const out: Record<string, LimitConfig> = {};
    limits.forEach((v, k) => { out[k] = { ...v }; });
    return out;
  }

  function clear() {
    limits.clear();
  }

  function check(i: CheckInput): CheckResult {
    const viols: Violation[] = [];
    const symbol = i.symbol || "";
    const eq = num(i.equity, equity);
    const gross = num(i.gross, 0);
    const realized = num(i.realized, 0);
    const peak = num(i.peakEquity, eq);
    const ordersMin = num(i.ordersLastMin, 0);

    // check helper
    function chk(key: string, cfg: LimitConfig) {
      if (!cfg) return;
      if (cfg.maxPosition != null && i.qty != null && Math.abs(i.qty) > cfg.maxPosition) {
        viols.push({ key, field: "maxPosition", value: i.qty, limit: cfg.maxPosition });
      }
      if (cfg.maxNotional != null && i.notional != null && Math.abs(i.notional) > cfg.maxNotional) {
        viols.push({ key, field: "maxNotional", value: i.notional, limit: cfg.maxNotional });
      }
      if (cfg.maxLoss != null && realized < cfg.maxLoss) {
        viols.push({ key, field: "maxLoss", value: realized, limit: cfg.maxLoss });
      }
      if (cfg.maxDrawdown != null && peak > 0) {
        const dd = (peak - eq) / peak;
        if (dd > cfg.maxDrawdown) {
          viols.push({ key, field: "maxDrawdown", value: dd, limit: cfg.maxDrawdown });
        }
      }
      if (cfg.maxGrossExposure != null && gross > cfg.maxGrossExposure) {
        viols.push({ key, field: "maxGrossExposure", value: gross, limit: cfg.maxGrossExposure });
      }
      if (cfg.maxLeverage != null && eq > 0) {
        const lev = gross / eq;
        if (lev > cfg.maxLeverage) {
          viols.push({ key, field: "maxLeverage", value: lev, limit: cfg.maxLeverage });
        }
      }
      if (cfg.maxOrdersPerMin != null && ordersMin > cfg.maxOrdersPerMin) {
        viols.push({ key, field: "maxOrdersPerMin", value: ordersMin, limit: cfg.maxOrdersPerMin });
      }
    }

    // symbol-level
    if (symbol && limits.has(symbol)) chk(symbol, limits.get(symbol)!);
    // global-level
    if (limits.has("global")) chk("global", limits.get("global")!);

    return { ok: viols.length === 0, violations: viols };
  }

  return { set, get, all, check, clear };
}

/* ---------------------------- helpers ---------------------------- */

function num(v: any, d: number) {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}