// frontend/types/Risk.ts
// Strong types for risk caps, runtime state, breaches, and helper formatters.

export type ISODate = string;
export type ISODateTime = string;

/** Matches backend LimitsConfig (risk/risk_limits.py). */
export interface RiskCaps {
  maxGrossUSD: number;
  maxNetUSD: number;
  maxPerNameUSD: number;
  maxDailyTurnoverUSD: number;
  maxTicketUSD: number;
  minTicketUSD: number;
  allowShort: boolean;
  maxWeightPerName?: number;   // 0..1
  maxNamesPerBucket?: number;
  maxParticipationRate?: number;
  minAdvUSD?: number;
  maxGamma?: number;
  maxVega?: number;
  maxTheta?: number;
  varLimitUSD?: number;
  maxDailyLossUSD?: number;
}

/** Runtime risk state (synced from backend RiskState). */
export interface RiskState {
  tsEpoch: number;
  positions: Record<string, number>;   // baseSymbol → signed USD notional
  tradedTodayUSD: number;
  dailyPnlUSD: number;
  lastActionTs: Record<string, number>;
  breaches: RiskBreach[];
}

/** Breach identifiers (keep in sync with risk_limits.py breached()). */
export type RiskBreach =
  | "MAX_GROSS"
  | "MAX_NET"
  | "DAILY_LOSS_LIMIT"
  | "CONCENTRATION"
  | "VAR"
  | "TURNOVER"
  | "PER_NAME_CAP"
  | "LIQUIDITY"
  | "GREEKS_CAP"
  | "OUT_OF_WINDOW"
  | "COOLDOWN";

/** Higher-level policy actions (risk_policies.py). */
export type PolicyAction = "APPROVE" | "QUEUE" | "REJECT" | "HALT";

/** Result of a policy/risk evaluation on an order batch. */
export interface RiskCheckResult {
  approved: any[]; // TradeOrder[]
  queued: any[];
  rejected: any[];
  halted: any[];
}

/** Light diagnostic entry for UI. */
export interface RiskEvent {
  asOf: ISODateTime;
  sid: string;
  action: PolicyAction;
  reason: string;
  details?: Record<string, any>;
}

/* ----------------------------------------------------------------------------------
 * Helpers for UI formatting
 * ---------------------------------------------------------------------------------- */

export const fmtUSD = (v: number, digits = 0) =>
  (v ?? 0).toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: digits });

export const fmtSignedUSD = (v: number, digits = 0) => {
  const s = fmtUSD(Math.abs(v ?? 0), digits);
  return (v ?? 0) >= 0 ? `+${s}` : `-${s}`;
};

export const pct = (v: number, digits = 1) =>
  `${(100 * (v ?? 0)).toFixed(digits)}%`;

/** Color-coded breach labels for dashboards. */
export function breachLabel(b: RiskBreach): { label: string; color: string } {
  switch (b) {
    case "MAX_GROSS": return { label: "Gross Cap", color: "red" };
    case "MAX_NET": return { label: "Net Cap", color: "red" };
    case "DAILY_LOSS_LIMIT": return { label: "Daily Loss", color: "orange" };
    case "CONCENTRATION": return { label: "Concentration", color: "amber" };
    case "VAR": return { label: "VaR", color: "purple" };
    case "TURNOVER": return { label: "Turnover", color: "blue" };
    case "PER_NAME_CAP": return { label: "Per-Name Cap", color: "blue" };
    case "LIQUIDITY": return { label: "Liquidity", color: "teal" };
    case "GREEKS_CAP": return { label: "Greeks Cap", color: "indigo" };
    case "OUT_OF_WINDOW": return { label: "Trading Window", color: "gray" };
    case "COOLDOWN": return { label: "Cooldown", color: "gray" };
    default: return { label: b, color: "gray" };
  }
}

/* ----------------------------------------------------------------------------------
 * Defaults & factories
 * ---------------------------------------------------------------------------------- */

export const DEFAULT_RISK_CAPS: RiskCaps = {
  maxGrossUSD: 25_000_000,
  maxNetUSD: 10_000_000,
  maxPerNameUSD: 5_000_000,
  maxDailyTurnoverUSD: 10_000_000,
  maxTicketUSD: 5_000_000,
  minTicketUSD: 5_000,
  allowShort: true,
  maxWeightPerName: 0.25,
  maxParticipationRate: 0.15,
  minAdvUSD: 1_000_000,
  varLimitUSD: 500_000,
  maxDailyLossUSD: 750_000,
};

export function makeRiskState(): RiskState {
  return {
    tsEpoch: Date.now() / 1000,
    positions: {},
    tradedTodayUSD: 0,
    dailyPnlUSD: 0,
    lastActionTs: {},
    breaches: [],
  };
}